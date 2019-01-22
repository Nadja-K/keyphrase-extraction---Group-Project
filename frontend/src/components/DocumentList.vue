<template>
    <v-list
        three-line
        subheader
        v-if="documents.length > 0"
    >
        <template v-for="(item, index) in documents">
            <v-list-tile
                :to="`${dataset}/document/${item.id}`"
                :key="item.id"
            >
                <v-list-tile-content>
                    <v-list-tile-title>{{ item.headline }}</v-list-tile-title>
                    <v-list-tile-sub-title>{{ item.text }}</v-list-tile-sub-title>
                </v-list-tile-content>
            </v-list-tile>
            <v-divider v-if="index + 1 < documents.length" :key="`divider-${index}`"></v-divider>
        </template>
    </v-list>
</template>

<script>
import { mapState } from 'vuex';

export default {
    computed: {
        ...mapState('documents', [
            'dataset',
            'documents',
        ]),
    },
};
</script>

<style scoped>

</style>
